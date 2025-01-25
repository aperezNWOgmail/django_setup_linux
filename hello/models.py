from django.db import models
from django.db import connection

# Create your models here.
class LogModel(models.Model):
    PageName = models.CharField(max_length=100);

    #formatted_string = "I have 100%% confidence." % ()
    #sql_query        = "SELECT AL.[ID_column],AL.[PageName],AL.[AccessDate],AL.[IpValue] FROM accessLogs AL WHERE AL.[LogType] = 1 AND  (AL.PAGENAME LIKE '%DEMO%' and AL.PAGENAME LIKE '%PAGE%') AND (AL.PAGENAME NOT LIKE '%ERROR%') AND AL.PAGENAME  NOT LIKE '%PAGE_DEMO_INDEX%' AND  UPPER(AL.PAGENAME) NOT LIKE '%CACHE%' AND AL.IPVALUE <> '::1'  order by  AL.[ID_column] desc".format()

    
    #with connection.cursor() as cursor:
    #    cursor.execute(sql_query)
    #    rows = cursor.fetchall();

    #for row in rows:
    #    print(row);

